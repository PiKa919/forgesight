import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-primary text-primary-foreground shadow hover:bg-primary/80",
        secondary:
          "border-transparent bg-zinc-800 text-zinc-300 hover:bg-zinc-700/80",
        destructive:
          "border-red-500/20 bg-red-500/15 text-red-400 shadow-sm hover:bg-red-500/25",
        outline:
          "border-zinc-800 text-zinc-300",
        success:
          "border-emerald-500/20 bg-emerald-500/15 text-emerald-400 shadow-sm hover:bg-emerald-500/25",
        warning:
          "border-amber-500/20 bg-amber-500/15 text-amber-400 shadow-sm hover:bg-amber-500/25",
        info:
          "border-sky-500/20 bg-sky-500/15 text-sky-400 shadow-sm hover:bg-sky-500/25",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

const Badge = React.forwardRef<HTMLDivElement, BadgeProps>(
  ({ className, variant, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(badgeVariants({ variant }), className)}
        {...props}
      />
    );
  }
);
Badge.displayName = "Badge";

export { Badge, badgeVariants };
